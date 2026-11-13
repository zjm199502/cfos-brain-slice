// 把 使用说明书.md 转成 Word。用法：node build.js <输出.docx> <字体>
const fs = require("fs");
const path = require("path");
const {
  Document, Packer, Paragraph, TextRun, HeadingLevel, AlignmentType, ImageRun, Table, TableRow, TableCell,
  WidthType, ShadingType, BorderStyle, LevelFormat, PageBreak, Footer, Header, PageNumber, ExternalHyperlink,
  TableLayoutType, InternalHyperlink, Bookmark,
} = require("docx");

const OUT = process.argv[2];
const FONT = process.argv[3] || "Microsoft YaHei";
const ROOT = "/home/user/manual";
const md = fs.readFileSync(path.join(ROOT, "使用说明书.md"), "utf8").split("\n");
const dims = JSON.parse(fs.readFileSync(path.join(ROOT, "build/dims.json"), "utf8"));

const PAGE_W = 11906, MARGIN = 1134, CONTENT = PAGE_W - 2 * MARGIN; // A4
const MAXW_PX = 620, MAXH_PX = 700;
const ACCENT = "1F5C99", TIP_BG = "E8F1FB", WARN_BG = "FFF4E0", HEAD_BG = "DCE8F5";
const f = (o = {}) => ({ font: { ascii: FONT, hAnsi: FONT, eastAsia: FONT, cs: FONT }, ...o });

// ---------- 行内格式：**粗体**、`代码`、<链接>、「按钮」
function runs(text, base = {}) {
  const out = [];
  const re = /(\*\*[^*]+\*\*|`[^`]+`|<https?:\/\/[^>]+>)/g;
  let last = 0, m;
  while ((m = re.exec(text))) {
    if (m.index > last) out.push(new TextRun(f({ text: text.slice(last, m.index), ...base })));
    const t = m[0];
    if (t.startsWith("**")) out.push(new TextRun(f({ text: t.slice(2, -2), bold: true, ...base })));
    else if (t.startsWith("`")) out.push(new TextRun({ text: t.slice(1, -1), font: { ascii: "Consolas", hAnsi: "Consolas", eastAsia: FONT }, shading: { type: ShadingType.CLEAR, fill: "EEEEEE", color: "auto" }, ...base }));
    else {
      const url = t.slice(1, -1);
      out.push(new ExternalHyperlink({ link: url, children: [new TextRun(f({ text: url, style: "Hyperlink", ...base }))] }));
    }
    last = m.index + t.length;
  }
  if (last < text.length) out.push(new TextRun(f({ text: text.slice(last), ...base })));
  return out;
}

const children = [];
const numberingConfigs = [];
let figNo = 0, chapter = 0, tocEntries = [];
const border = { style: BorderStyle.SINGLE, size: 4, color: "B7C7D9" };
const borders = { top: border, bottom: border, left: border, right: border };

function table(rows) {
  const ncol = rows[0].length;
  // 列宽：按每列最长文字估计，限制在合理范围
  const len = Array(ncol).fill(0);
  rows.forEach(r => r.forEach((c, i) => { len[i] = Math.max(len[i], [...c.replace(/\*\*|`/g, "")].length); }));
  const minw = len.map(l => Math.min(l, 7) * 200 + 260);           // 短列至少能放下一行文字
  const extra = CONTENT - minw.reduce((a, b) => a + b, 0);
  const want = len.map(l => Math.max(0, Math.min(l, 50) - 7));
  const wt = want.reduce((a, b) => a + b, 0) || 1;
  let w = minw.map((m, i) => Math.floor(m + Math.max(extra, 0) * want[i] / wt));
  w[ncol - 1] += CONTENT - w.reduce((a, b) => a + b, 0);
  return new Table({
    width: { size: CONTENT, type: WidthType.DXA }, columnWidths: w, layout: TableLayoutType.FIXED,
    rows: rows.map((r, ri) => new TableRow({
      tableHeader: ri === 0, cantSplit: true,
      children: r.map((c, ci) => new TableCell({
        borders, width: { size: w[ci], type: WidthType.DXA },
        shading: ri === 0 ? { type: ShadingType.CLEAR, fill: HEAD_BG, color: "auto" } : undefined,
        margins: { top: 60, bottom: 60, left: 100, right: 100 },
        children: [new Paragraph({ spacing: { before: 0, after: 0, line: 300 }, children: runs(c, ri === 0 ? { bold: true, size: 20 } : { size: 20 }) })],
      })),
    })),
  });
}

function image(ref, caption) {
  const d = dims[ref];
  if (!d) { console.error("no image", ref); return; }
  let w = MAXW_PX, h = Math.round(d.h * MAXW_PX / d.w);
  if (d.w < MAXW_PX) { w = d.w; h = d.h; }
  if (h > MAXH_PX) { w = Math.round(w * MAXH_PX / h); h = MAXH_PX; }
  figNo++;
  children.push(new Paragraph({ alignment: AlignmentType.CENTER, keepNext: true, spacing: { before: 120, after: 60 },
    children: [new ImageRun({ type: d.type, data: fs.readFileSync(d.file), transformation: { width: w, height: h },
      altText: { title: caption, description: caption, name: path.basename(d.file) } })] }));
  children.push(new Paragraph({ alignment: AlignmentType.CENTER, spacing: { before: 0, after: 200 },
    children: runs(`图 ${figNo}　${caption}`, { size: 18, color: "555555" }) }));
}

function callout(text) {
  const warn = text.startsWith("注意");
  const label = warn ? "⚠ " : "💡 ";
  const m = text.match(/^(提示|注意)：(.*)$/);
  const parts = m ? [new TextRun(f({ text: label + m[1] + "：", bold: true, color: warn ? "B35900" : ACCENT })), ...runs(m[2])] : runs(text);
  children.push(new Paragraph({
    spacing: { before: 120, after: 160, line: 340 }, indent: { left: 200, right: 200 },
    shading: { type: ShadingType.CLEAR, fill: warn ? WARN_BG : TIP_BG, color: "auto" },
    border: { left: { style: BorderStyle.SINGLE, size: 24, color: warn ? "E69500" : ACCENT, space: 8 } },
    children: parts,
  }));
}

// ---------- 封面
const titleLines = md.filter(l => l.startsWith("% ")).map(l => l.slice(2));
children.push(new Paragraph({ spacing: { before: 3000, after: 400 }, alignment: AlignmentType.CENTER,
  children: [new TextRun(f({ text: "cFos 脑片分析", bold: true, size: 64, color: ACCENT }))] }));
children.push(new Paragraph({ spacing: { after: 1200 }, alignment: AlignmentType.CENTER,
  children: [new TextRun(f({ text: "软件使用说明书", bold: true, size: 48 }))] }));
const sub = (titleLines[1] || "").split("·").map(s => s.trim());
sub.forEach(s => children.push(new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 160 },
  children: [new TextRun(f({ text: s, size: 28, color: "444444" }))] })));
children.push(new Paragraph({ spacing: { before: 2400 }, alignment: AlignmentType.CENTER,
  children: [new TextRun(f({ text: "小鼠脑切片荧光图像：细胞计数 · 共标 · CCFv3 脑图谱配准 · 按脑区导出 Excel", size: 22, color: "666666" }))] }));

// ---------- 目录（静态，点击可跳转）
const tocIdx = children.length;
const chapters = md.filter(l => /^# /.test(l)).map(l => l.slice(2).trim());
children.push(new Paragraph({ children: [new PageBreak()] }));
children.push(new Paragraph({ heading: HeadingLevel.TITLE, children: [new TextRun(f({ text: "目录", bold: true, size: 36, color: ACCENT }))], spacing: { after: 300 } }));
chapters.forEach((c, i) => children.push(new Paragraph({ spacing: { after: 80 }, indent: { left: 200 },
  children: [new InternalHyperlink({ anchor: `ch${i}`, children: [new TextRun(f({ text: c, size: 24 }))] })] })));
// 每章下面的节
// （放在章名下面一行，小字）
// 为简洁起见只列章

// ---------- 正文
let i = 0, chIdx = 0;
while (i < md.length) {
  let l = md[i];
  if (l.startsWith("% ") || l.trim() === "") { i++; continue; }
  if (l.startsWith("# ")) {
    children.push(new Paragraph({ children: [new PageBreak()] }));
    children.push(new Paragraph({ heading: HeadingLevel.HEADING_1, children: [new Bookmark({ id: `ch${chIdx++}`, children: [new TextRun(f({ text: l.slice(2).trim() }))] })] }));
    i++; continue;
  }
  if (l.startsWith("## ")) { children.push(new Paragraph({ heading: HeadingLevel.HEADING_2, children: [new TextRun(f({ text: l.slice(3).trim() }))] })); i++; continue; }
  if (l.startsWith("### ")) { children.push(new Paragraph({ heading: HeadingLevel.HEADING_3, children: [new TextRun(f({ text: l.slice(4).trim() }))] })); i++; continue; }
  if (l.startsWith("![")) { const m = l.match(/!\[([^\]]*)\]\(([^)]+)\)/); image(m[2], m[1]); i++; continue; }
  if (l.startsWith("> ")) { callout(l.slice(2).trim()); i++; continue; }
  if (l.startsWith("|")) {
    const rows = [];
    while (i < md.length && md[i].startsWith("|")) {
      const cells = md[i].trim().replace(/^\||\|$/g, "").split("|").map(s => s.trim());
      if (!cells.every(c => /^-+$/.test(c))) rows.push(cells);
      i++;
    }
    children.push(table(rows));
    children.push(new Paragraph({ spacing: { before: 0, after: 120 }, children: [] }));
    continue;
  }
  if (/^- /.test(l)) {
    while (i < md.length && /^- /.test(md[i])) {
      children.push(new Paragraph({ numbering: { reference: "bullets", level: 0 }, spacing: { after: 60, line: 340 }, children: runs(md[i].slice(2)) }));
      i++;
      while (i < md.length && /^ {2,}- /.test(md[i])) {
        children.push(new Paragraph({ numbering: { reference: "bullets", level: 1 }, spacing: { after: 40, line: 340 }, children: runs(md[i].trim().slice(2)) }));
        i++;
      }
    }
    continue;
  }
  if (/^\d+\. /.test(l)) {
    const ref = `num${i}`;
    numberingConfigs.push(ref);
    while (i < md.length && (/^\d+\. /.test(md[i]) || /^ {2,}\S/.test(md[i]) || (md[i].trim() === "" && i + 1 < md.length && /^ {2,}\S/.test(md[i + 1])))) {
      const s = md[i];
      if (/^\d+\. /.test(s)) children.push(new Paragraph({ numbering: { reference: ref, level: 0 }, spacing: { after: 60, line: 340 }, children: runs(s.replace(/^\d+\. /, "")) }));
      else if (/^ {2,}- /.test(s)) children.push(new Paragraph({ numbering: { reference: "bullets", level: 1 }, spacing: { after: 40, line: 340 }, children: runs(s.trim().slice(2)) }));
      else if (s.trim() !== "") children.push(new Paragraph({ indent: { left: 720 }, spacing: { after: 60, line: 340 }, children: runs(s.trim()) }));
      i++;
    }
    continue;
  }
  // 普通段落；“问：”“答：”特殊处理
  if (/^\*\*问：/.test(l)) {
    children.push(new Paragraph({ keepNext: true, spacing: { before: 200, after: 40 }, children: runs(l, { color: ACCENT }) }));
    i++; continue;
  }
  children.push(new Paragraph({ spacing: { after: 120, line: 360 }, children: runs(l) }));
  i++;
}


const numLevels = (fmt) => [0, 1].map(lv => ({ level: lv, format: fmt, text: fmt === LevelFormat.BULLET ? (lv ? "◦" : "•") : `%${lv + 1}.`,
  alignment: AlignmentType.LEFT, style: { paragraph: { indent: { left: 720 * (lv + 1) - 360, hanging: 360 } }, run: { font: FONT } } }));
const doc = new Document({
  creator: "cFos 脑片分析", title: "cFos 脑片分析 软件使用说明书 v0.7",
  styles: {
    default: { document: { run: { font: { ascii: FONT, hAnsi: FONT, eastAsia: FONT, cs: FONT }, size: 22 } } },
    paragraphStyles: [
      { id: "Title", name: "Title", basedOn: "Normal", run: { font: FONT, size: 36, bold: true }, paragraph: { spacing: { before: 0, after: 200 }, keepNext: true } },
      { id: "Heading1", name: "Heading 1", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { font: FONT, size: 36, bold: true, color: ACCENT }, paragraph: { spacing: { before: 0, after: 300 }, keepNext: true, outlineLevel: 0,
          border: { bottom: { style: BorderStyle.SINGLE, size: 12, color: ACCENT, space: 6 } } } },
      { id: "Heading2", name: "Heading 2", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { font: FONT, size: 28, bold: true, color: "1A1A1A" }, paragraph: { spacing: { before: 360, after: 160 }, keepNext: true, outlineLevel: 1 } },
      { id: "Heading3", name: "Heading 3", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { font: FONT, size: 24, bold: true }, paragraph: { spacing: { before: 240, after: 120 }, keepNext: true, outlineLevel: 2 } },
    ],
  },
  numbering: { config: [{ reference: "bullets", levels: numLevels(LevelFormat.BULLET) },
    ...numberingConfigs.map(r => ({ reference: r, levels: numLevels(LevelFormat.DECIMAL) }))] },
  sections: [{
    properties: { page: { size: { width: PAGE_W, height: 16838 }, margin: { top: 1300, bottom: 1300, left: MARGIN, right: MARGIN } }, titlePage: true },
    headers: { default: new Header({ children: [new Paragraph({ alignment: AlignmentType.RIGHT, children: [new TextRun(f({ text: "cFos 脑片分析 使用说明书 v0.7", size: 16, color: "888888" }))] })] }) },
    footers: { default: new Footer({ children: [new Paragraph({ alignment: AlignmentType.CENTER, children: [new TextRun(f({ children: [PageNumber.CURRENT], size: 18, color: "666666" }))] })] }) },
    children,
  }],
});
Packer.toBuffer(doc).then(b => { fs.writeFileSync(OUT, b); console.log("written", OUT, b.length, "figures", figNo); });

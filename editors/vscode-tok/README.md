# Tok Language for VS Code

Syntax highlighting for the [Tok](https://github.com/user/tok-lang) programming language.

## Features

- Syntax highlighting for `.tok` files
- String interpolation highlighting (`"hello {name}"`)
- Raw string support (`` `backtick strings` ``)
- Bracket matching and auto-closing
- Comment toggling (`#` and `//`)

## Highlighting Covers

- Keywords: `f`, `go`, `sel`, loop (`~`), return (`^`), break (`!`), continue (`>!`)
- Constants: `T`, `F`, `N`
- Imports: `@"math"`, `@"str"`, etc.
- Lambdas: `\(x) = ...`
- All operators: pipeline (`|>`), filter (`?>`), reduce (`/>`), channels (`<-`), etc.
- Type annotations (`:i`, `:f`, `:s`, etc.)
- Builtin functions (`p`, `pl`, `sort`, `push`, `type`, etc.)
- Number literals (decimal, hex, binary, octal, float, scientific)

## Installation

### From source

1. Copy or symlink this folder into your VS Code extensions directory:

```sh
# macOS / Linux
ln -s /path/to/tok-lang/editors/vscode-tok ~/.vscode/extensions/tok-lang

# Or copy
cp -r /path/to/tok-lang/editors/vscode-tok ~/.vscode/extensions/tok-lang
```

2. Restart VS Code
3. Open any `.tok` file

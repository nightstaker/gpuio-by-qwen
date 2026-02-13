# Contributing to gpuio

Thank you for your interest in contributing to gpuio! This document outlines the process for contributing to this GPU-initiated IO accelerator project.

## Code of Conduct

Be respectful, collaborative, and constructive. We welcome contributions from everyone regardless of experience level.

## How to Contribute

### 1. Report Bugs

- Open an issue with a clear title and description
- Include steps to reproduce
- Specify your environment (OS, compiler, GPU model)

### 2. Suggest Enhancements

- Provide a clear description of the feature
- Explain the use case and benefits
- If possible, propose an implementation approach

### 3. Submit Code

#### Branch Naming

Use the following convention:
- `task/1-2-docs-generation` — Task from tasklist (Phase-Number-Description)
- `feature/your-feature` — New features
- `fix/bug-description` — Bug fixes
- `refactor/what-youre-doing` — Code refactoring

#### Commit Messages

Follow the Conventional Commits format:
```
type: short description

Longer explanation if needed
```

Types: `feat`, `fix`, `docs`, `refactor`, `test`, `build`, `chore`

Examples:
- `feat: add MemIO engine skeleton`
- `fix: correct memory leak in context_destroy`
- `docs: add API documentation for gpuio_init`

#### Pull Request Process

1. Fork the repository
2. Create your feature branch (`git checkout -b task/1-2-docs-generation`)
3. Commit your changes (`git commit -m 'feat: add documentation'`)
4. Push to your branch (`git push origin task/1-2-docs-generation`)
5. Open a Pull Request

#### PR Requirements

- All CI checks must pass
- All tests must pass (`ctest`)
- Code must compile without warnings (`-Werror`)
- Documentation updated if needed
- Link to the relevant task in tasklist.md

### 4. Review Process

- At least one maintainer review required
- Address review comments promptly
- Maintainer will merge once CI passes and approvals received

## Development Setup

```bash
git clone https://github.com/nightstaker/gpuio-by-qwen.git
cd gpuio-by-qwen
mkdir -p build && cd build
cmake .. -DBUILD_TESTS=ON
make -j$(nproc)
ctest --output-on-failure
```

## Questions?

Open an issue or reach out to the maintainers.

## Recognition

Contributors are acknowledged in:
- Commit history
- Release notes
- Project documentation
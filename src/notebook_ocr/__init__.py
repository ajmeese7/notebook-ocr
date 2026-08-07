"""Photos-to-text pipeline: transcribe handwritten notebook pages with the Claude API."""

# Register the HEIF/HEIC opener with Pillow so iPhone photos load transparently.
try:
    from pillow_heif import register_heif_opener

    register_heif_opener()
except ImportError:  # pillow-heif is a declared dependency; guard for stripped installs.
    pass

from langchain_core.documents import Document

chunk = Document(
    page_content=('- 말합니다.\n'
 '- 50 -제2관 보험금의 지급제 3조 (보험금의 지급사유)각 특별약관의 보장을 따릅니다.제 4조 (보험금 지급에 관한 세부규정)# 각 '
 '특별약관의 보장을 따릅니다.# 제 5조 (보험료 납입면제)보험료 납입면제 사항은 기본계약의 보험료 납입면제 사항을 준용합니다.# 제 6조 '
 '(보험료 납입면제에 관한 세부규정)보험료 납입면제에 관한 세부규정은 기본계약의 보험료 납입면제에 관한 세부규정을 준'),
    metadata={'source_doc': {'total_pages': 129},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_5_pet_majestic_cat_31_47_50_159.pdf',
         'insurer_code': 'samsung',
         'product_code': '5',
         'product_name': '(장기)무배당 삼성화재 펫보험 의기냥냥',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown'},
)

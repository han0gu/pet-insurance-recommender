from langchain_core.documents import Document

chunk = Document(
    page_content=('- 4. 핵연료물질 또는 핵연료물질에 의하여 오염된 물질의 방사성, 폭발성 또는 그\n'
 '- 밖의 유해한 특성에 의한 사고\n'
 '- 부 가 설 명\n'
 '| ∙ 핵연료물질 : 사용된 | 연료를 포함합니다. |\n'
 '| --- | --- |\n'
 '| ∙ 핵연료물질에 의하여 오염된 물질 : 원자핵 분열 생성물을 포함합니다. 제4호 이외에 방사선을 쬐는 것 또는 방사능 오염 | ∙ '
 '핵연료물질에 의하여 오염된 물질 : 원자핵 분열 생성물을 포함합니다. 제4호 이외에 방사선을 쬐는 것 또는 방사능 오염 |'),
    metadata={'source_doc': {'total_pages': 113},
 'doc': {'doc_type': 'terms',
         'file_name': 'kb_17390_gold_pet_dog_54_72_74_166.pdf',
         'insurer_code': 'kb',
         'product_code': '17390',
         'product_name': '[일반보험] KB반려행복펫보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown'},
)

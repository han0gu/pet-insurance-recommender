from langchain_core.documents import Document

chunk = Document(
    page_content=('① 회사는 보통약관 제5조(보험금을 지급하지 않은 사유) 제2항 제4호에도 불구하고 비뇨기계질환, 전염성복막염 또는 기타 이들과 유사한 '
 '질병 또는 상해를 원인으로 하여 그 치료를 직접적인 목적으로 동물병원에 통원 또는 입원하여 수의사에게 치료를 받은 때에는 피보험자가 '
 '부담한 반려동물의 치료비를 보통약관 제4조(보험금의 지급사유)에 따라 피보험자에게 치료비보험금으로 보상하여 드립니다. 단, 동물병원에서 '
 '수의사에게 수술을 받은 경우 수술 당일 발생한 수술비 및 치료비는 보상하여 드리지 않습니다'),
    metadata={'source_doc': {'total_pages': 50},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_1_maum_pet_1_21_22_50.pdf',
         'insurer_code': 'meritz',
         'product_code': '1',
         'product_name': '메리츠 마음든든 반려동물보험',
         'total_pages': 50,
         'page': 32},
 'term_type': 'special',
 'clause': {'clause_type': 'exclusion', 'risk_domains': ['urinary']},
 'indexing': {'chunk_id': 'chunk_000191',
              'chunk_char_len': 276,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'rule',
              'tag_confidence': 0.85}},
)

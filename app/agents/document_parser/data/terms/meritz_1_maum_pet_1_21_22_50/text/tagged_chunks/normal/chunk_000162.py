from langchain_core.documents import Document

chunk = Document(
    page_content=('비뇨기계질환, 전염성복막염 또는 기타 이들과 유사한 질병 또는 상해를 원인으로 하여\n'
 '그 치료를 직접적인 목적으로 동물병원에 통원 또는 입원하여 수의사에게 치료를 받은\n'
 '때에는 피보험자가 부담한 반려동물의 치료비를 보통약관 제4조(보험금의 지급사유)에\n'
 '따라 피보험자에게 치료비보험금으로 보상하여 드립니다. 단, 동물병원에서 수의사에게\n'
 '수술을 받은 경우 수술 당일 발생한 수술비 및 치료비는 보상하여 드리지 않습니다.\n'
 '② 제1항에도 불구하고 보험개시일로부터 그 날을 포함하여 보험증권에 기재된'),
    metadata={'source_doc': {'total_pages': 50},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_1_maum_pet_1_21_22_50.pdf',
         'insurer_code': 'meritz',
         'product_code': '1',
         'product_name': '메리츠 마음든든 반려동물보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'coverage', 'risk_domains': ['urinary']},
 'indexing': {'chunk_id': 'chunk_000162',
              'chunk_char_len': 272,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.55}},
)

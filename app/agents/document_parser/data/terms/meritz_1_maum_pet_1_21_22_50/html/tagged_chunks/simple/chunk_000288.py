from langchain_core.documents import Document

chunk = Document(
    page_content=("보장 특별약관</h1><p id='7' data-category='paragraph' "
 "style='font-size:14px'>제1조(보상하는 손해)</p><br><p id='8' data-category='list' "
 "style='font-size:14px'>① 회사는 보통약관 제5조(보험금을 지급하지 않은 사유) 제2항 제4호에도 "
 '불구하고<br>비뇨기계질환, 전염성복막염 또는 기타 이들과 유사한 질병 또는 상해를 원인으로 하여<br>그 치료를 직접적인 목적으로 '
 '동물병원에 통원 또는 입원하여 수의사에게 치료를 받은<br>때에는'),
    metadata={'source_doc': {'total_pages': 50},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_1_maum_pet_1_21_22_50.pdf',
         'insurer_code': 'meritz',
         'product_code': '1',
         'product_name': '메리츠 마음든든 반려동물보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'exclusion',
            'risk_domains': ['digestive', 'urinary']},
 'indexing': {'chunk_id': 'chunk_000288',
              'chunk_char_len': 297,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.85}},
)

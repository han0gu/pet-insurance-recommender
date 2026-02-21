from langchain_core.documents import Document

chunk = Document(
    page_content=(". 다만, 손해가 그 가족의 고의로 인하여<br>발생한 경우에는 그 권리를 취득합니다.</p><h1 id='87' "
 "style='font-size:14px'>제15조(계약 후 알릴 의무)</h1><br><p id='88' "
 "data-category='paragraph' style='font-size:14px'>① 계약을 맺은 후 보험의 목적에 아래와 같은 "
 '사실이 생긴 경우에는 계약자나 피보험자는<br>지체없이 서면으로 회사에 알리고 보험증권에 확인을 받아야 합니다.</p><br><p '
 "id='89' data-category='list'"),
    metadata={'source_doc': {'total_pages': 50},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_1_maum_pet_1_21_22_50.pdf',
         'insurer_code': 'meritz',
         'product_code': '1',
         'product_name': '메리츠 마음든든 반려동물보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'other', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000257',
              'chunk_char_len': 299,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)

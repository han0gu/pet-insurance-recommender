from langchain_core.documents import Document

chunk = Document(
    page_content=("손해를 법적으로 보상해 주기 위해서 법원에 납부하<br>는 공탁금을 대신하는 보험상품의 보험료를 말합니다.</p><h1 id='31' "
 "style='font-size:14px'>제4조(보상하지 않는 손해)</h1><br><p id='32' "
 "data-category='paragraph' style='font-size:14px'>회사는 아래의 사유를 원인으로 하여 생긴 "
 "배상책임을 부담함으로써 입은 손해는 보상하지<br>않습니다.</p><br><p id='33' data-category='list'"),
    metadata={'source_doc': {'total_pages': 50},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_1_maum_pet_1_21_22_50.pdf',
         'insurer_code': 'meritz',
         'product_code': '1',
         'product_name': '메리츠 마음든든 반려동물보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'exclusion', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000212',
              'chunk_char_len': 277,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.85}},
)

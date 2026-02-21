from langchain_core.documents import Document

chunk = Document(
    page_content=('‘계약자’라 합니다)와 보험회사(이<br>하 ‘회사’라 합니다) 사이에 보험증권에 기재된 반려동물의 질병 또는 상해로 인하여 '
 "피보<br>험자가 입을 손해에 대한 위험을 보장하기 위하여 체결됩니다.</p><h1 id='4' "
 "style='font-size:14px'>제2조(용어의 정의)</h1><br><p id='5' "
 "data-category='paragraph' style='font-size:14px'>이 계약에서 사용되는 용어의 정의는 이 계약의 "
 '다른 조항에서 달리 정의되지 않는 한 다음<br>과 같습니다.</p><br><h1'),
    metadata={'source_doc': {'total_pages': 50},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_1_maum_pet_1_21_22_50.pdf',
         'insurer_code': 'meritz',
         'product_code': '1',
         'product_name': '메리츠 마음든든 반려동물보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'coverage', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000001',
              'chunk_char_len': 297,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.8}},
)

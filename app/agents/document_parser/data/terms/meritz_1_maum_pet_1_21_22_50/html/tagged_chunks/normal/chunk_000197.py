from langchain_core.documents import Document

chunk = Document(
    page_content=("예금자보호법에서 정하는<br>바에 따라 그 지급을 보장합니다.</p><h1 id='10' "
 "style='font-size:14px'>【예금자보호제도】</h1><br><p id='11' "
 "data-category='paragraph' style='font-size:14px'>예금자보호제도란 예금보험공사가 평소에 금융기관으로 "
 '부터 보험료를 받아 기금을 적<br>립한 후, 금융기관이 영업정지나 파산 등으로 예금을 지급할 수 없게 되면 금융기관을<br>대신하여 '
 '예금을 지급하는 제도를 말합니다'),
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
 'indexing': {'chunk_id': 'chunk_000197',
              'chunk_char_len': 271,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.55}},
)

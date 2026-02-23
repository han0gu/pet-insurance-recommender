from langchain_core.documents import Document

chunk = Document(
    page_content=(". 피보험자 본인 또는 배우자와 생계를 같이 하는 동거 친족 및 별거 중인 미혼자녀</p><br><p id='22' "
 "data-category='paragraph' style='font-size:14px'>② 위 제1항에서 피보험자 본인과 본인 이외의 "
 "피보험자와의 관계는 사고발생 당시의 관<br>계를 말합니다.</p><footer id='23' "
 "style='font-size:14px'>제2관 보험금의 지급</footer><footer id='24' "
 "style='font-size:14px'>- 2 -</footer><h1 id='25'"),
    metadata={'source_doc': {'total_pages': 50},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_1_maum_pet_1_21_22_50.pdf',
         'insurer_code': 'meritz',
         'product_code': '1',
         'product_name': '메리츠 마음든든 반려동물보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'other', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000016',
              'chunk_char_len': 300,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)

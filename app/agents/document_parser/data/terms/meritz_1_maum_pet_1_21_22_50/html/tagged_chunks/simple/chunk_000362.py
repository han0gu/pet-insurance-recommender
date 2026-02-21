from langchain_core.documents import Document

chunk = Document(
    page_content=('. 다만, 다음 각 호의 보험료별로 그 합계액이 각각 연 100만원을 초과하는<br>경우 그 초과하는 금액은 각각 없는 것으로 '
 "한다.</p><br><p id='38' data-category='list' style='font-size:14px'>1. 기본공제대상자 "
 '중 장애인을 피보험자 또는 수익자로 하는 장애인전용보험으로서<br>대통령령으로 정하는 장애인전용보장성보험료<br>2'),
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
 'indexing': {'chunk_id': 'chunk_000362',
              'chunk_char_len': 212,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.8}},
)

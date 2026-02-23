from langchain_core.documents import Document

chunk = Document(
    page_content=('. 안락사 비용, 시체처치 및 해부검사, 장례비, 이장비 등 사후에 필요한 비용<br>19. 마이크로 칩 이식 비용, 각종 증빙서류의 '
 '작성비용(우송비 포함)<br>20. 과잉진료행위로 인한 비용<br>21. 아포퀠(Apoquel) 등의 JAK inhibitor(Janus '
 'kinase inhibitor) 약물<br>22'),
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
 'indexing': {'chunk_id': 'chunk_000046',
              'chunk_char_len': 178,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)

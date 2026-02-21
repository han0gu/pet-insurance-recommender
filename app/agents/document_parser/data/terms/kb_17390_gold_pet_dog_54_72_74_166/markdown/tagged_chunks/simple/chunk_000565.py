from langchain_core.documents import Document

chunk = Document(
    page_content=('- 이와 동종의 비용\n'
 '- 15. 안락사 비용, 시체처치 및 해부검사, 장례비, 이장비 등 사후에 필요한 비용\n'
 '- 16. 마이크로칩의 삽입비용, 각종 증빙서류의 작성비용(우송비 포함)\n'
 '- 17. 과잉진료행위로 인한 비용\n'
 '- 18. 아포퀠(Apoquel) 등의 JAK inhibitor(Janus kinase inhibitor) 약물과 사이토\n'
 '- 포인트(Cytopoint)\n'
 '19. 첩모난생(속눈썹 질환) 및 눈물샘 치료(누루관시술 등) 등의 안검 외·내반 및\n'
 '비루관 관련 질환으로 인한 비용'),
    metadata={'source_doc': {'total_pages': 113},
 'doc': {'doc_type': 'terms',
         'file_name': 'kb_17390_gold_pet_dog_54_72_74_166.pdf',
         'insurer_code': 'kb',
         'product_code': '17390',
         'product_name': '[일반보험] KB반려행복펫보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'other', 'risk_domains': ['digestive', 'eye']},
 'indexing': {'chunk_id': 'chunk_000565',
              'chunk_char_len': 272,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)

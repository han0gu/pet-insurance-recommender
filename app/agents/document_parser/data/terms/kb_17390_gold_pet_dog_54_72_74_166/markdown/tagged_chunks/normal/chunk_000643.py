from langchain_core.documents import Document

chunk = Document(
    page_content=('\uf000 제1항의 경우 반려동물장례비용지원금보장개시일은 계약일로부터 그날을 포함하\n'
 '여 30일이 지난날의 다음날로 합니다. 단, 계약일은 제1회 보험료를 받은 날로 합- \n'
 '| 니다. | 니다. |\n'
 '| --- | --- |\n'
 '| 예 시 반려동물장례비용지원금의 보장개시일 계약일 보장개시일 30일 | 예 시 반려동물장례비용지원금의 보장개시일 계약일 보장개시일 '
 '30일 |\n'
 '| 2024년 4월 10일 2024년 5월 9일 | 2024년 4월 10일 2024년 5월 9일 |'),
    metadata={'source_doc': {'total_pages': 113},
 'doc': {'doc_type': 'terms',
         'file_name': 'kb_17390_gold_pet_dog_54_72_74_166.pdf',
         'insurer_code': 'kb',
         'product_code': '17390',
         'product_name': '[일반보험] KB반려행복펫보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'coverage', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000643',
              'chunk_char_len': 255,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.55}},
)

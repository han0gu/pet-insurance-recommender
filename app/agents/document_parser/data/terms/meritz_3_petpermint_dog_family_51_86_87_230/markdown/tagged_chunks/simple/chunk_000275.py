from langchain_core.documents import Document

chunk = Document(
    page_content=('경우에는 적용하지 않습니다.121【슬관절탈구, 고관절탈구, 슬관절형성부전, 고관절형성\n'
 '부전(대퇴 골두 허혈성 괴사 포함) 등 보장예시】![image](/image/placeholder)\n'
 '- Chart Type: bar\n'
 '|  | 보장개시일 | 슬관절탈구 |\n'
 '| --- | --- | --- |\n'
 '| item_01 | 410 | 2026.4 |\n'
 '※ 설명\n'
 '보장개시일로부터 1년이내 발생한 슬관절탈구 : 보험금 미지급\uf000 제1항의「연간」이라 함은 계약일부터 매 1년 단위로 도\n'
 '래하는 계약해당일 전일까지의 기간을 말합니다.'),
    metadata={'source_doc': {'total_pages': 180},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_3_petpermint_dog_family_51_86_87_230.pdf',
         'insurer_code': 'meritz',
         'product_code': '3',
         'product_name': '무배당 펫퍼민트 Puppy&Family보험 다이렉트2601',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'coverage', 'risk_domains': ['digestive', 'joint']},
 'indexing': {'chunk_id': 'chunk_000275',
              'chunk_char_len': 286,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.8}},
)

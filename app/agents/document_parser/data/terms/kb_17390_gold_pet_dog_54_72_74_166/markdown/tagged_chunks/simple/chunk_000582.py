from langchain_core.documents import Document

chunk = Document(
    page_content=('예 시 1 반려동물주요치료 보장개시일\n'
 '계약일 보장개시일\n'
 '30일# 2024년 4월 10일2024년 5월 9일# - 단, 상해(상해로 인한 창상 또는 교상, 이물섭취를 포함)를 직접적인 원인으# '
 '로 치료를 받은 경우에는 보장개시일은 보험계약일로 합니다.# 예 시 2# 백내장/녹내장수술, 특정약물치료Ⅱ, '
 '항암약물치료![image](/image/placeholder)\n'
 '보장개시일\n'
 '계약일 보장개시일\n'
 '90일\n'
 '2024년 4월 10일 2024년 7월 9일- \uf000 제3항에서 "연간"이란 계약일로부터 매1년 단위로 도래하는 계약해당일 전일까지'),
    metadata={'source_doc': {'total_pages': 113},
 'doc': {'doc_type': 'terms',
         'file_name': 'kb_17390_gold_pet_dog_54_72_74_166.pdf',
         'insurer_code': 'kb',
         'product_code': '17390',
         'product_name': '[일반보험] KB반려행복펫보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'coverage', 'risk_domains': ['digestive', 'eye']},
 'indexing': {'chunk_id': 'chunk_000582',
              'chunk_char_len': 294,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.8}},
)

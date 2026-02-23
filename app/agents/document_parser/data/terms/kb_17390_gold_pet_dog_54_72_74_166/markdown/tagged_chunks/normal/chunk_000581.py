from langchain_core.documents import Document

chunk = Document(
    page_content=('- 여 90일 이내에 발생한 "백내장/녹내장수술", "특정약물치료Ⅱ", "항암약물치료\n'
 '- 112 -" 또는 기타 이들과 유사한 사고에 대해서는 보험금을 지급하지 않습니다.![image](/image/placeholder)\n'
 '예 시 1 반려동물주요치료 보장개시일\n'
 '계약일 보장개시일'),
    metadata={'source_doc': {'total_pages': 113},
 'doc': {'doc_type': 'terms',
         'file_name': 'kb_17390_gold_pet_dog_54_72_74_166.pdf',
         'insurer_code': 'kb',
         'product_code': '17390',
         'product_name': '[일반보험] KB반려행복펫보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'exclusion', 'risk_domains': ['digestive', 'eye']},
 'indexing': {'chunk_id': 'chunk_000581',
              'chunk_char_len': 154,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.85}},
)

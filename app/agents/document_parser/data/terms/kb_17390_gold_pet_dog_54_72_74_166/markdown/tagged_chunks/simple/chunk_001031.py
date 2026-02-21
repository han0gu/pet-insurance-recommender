from langchain_core.documents import Document

chunk = Document(
    page_content=('| 별표17 | 반려동물(강아지) | 특정 질병 분류표 |\n'
 '| --- | --- | --- |\n'
 '| 코드 801 | 특정 질병 근골격계질환 | 세부 질병명 |\n'
 '| 코드 801 | 특정 질병 근골격계질환 |  |\n'
 '| 코드 801 | 특정 질병 근골격계질환 | 고관절 탈구 / 아탈구 |\n'
 '| 코드 801 | 특정 질병 근골격계질환 | 고관절이형성증 골절 (골반) |\n'
 '| 코드 801 | 특정 질병 근골격계질환 | 골절 (기타) |\n'
 '| 코드 801 | 특정 질병 근골격계질환 | 골절 (전지) |'),
    metadata={'source_doc': {'total_pages': 113},
 'doc': {'doc_type': 'terms',
         'file_name': 'kb_17390_gold_pet_dog_54_72_74_166.pdf',
         'insurer_code': 'kb',
         'product_code': '17390',
         'product_name': '[일반보험] KB반려행복펫보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'other', 'risk_domains': ['joint']},
 'indexing': {'chunk_id': 'chunk_001031',
              'chunk_char_len': 273,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)

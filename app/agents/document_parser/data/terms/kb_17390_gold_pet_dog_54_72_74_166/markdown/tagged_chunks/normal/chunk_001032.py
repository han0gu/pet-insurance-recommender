from langchain_core.documents import Document

chunk = Document(
    page_content=('| 코드 801 | 특정 질병 근골격계질환 | 골절 (전지) |\n'
 '| 코드 801 | 특정 질병 근골격계질환 | 골절 (후지) |\n'
 '| 코드 801 | 특정 질병 근골격계질환 | 관절염 |\n'
 '| 코드 801 | 특정 질병 근골격계질환 | 근염 |\n'
 '| 코드 801 | 특정 질병 근골격계질환 | 대퇴골두허혈성괴사 |\n'
 '| 코드 801 | 특정 질병 근골격계질환 | 생후 골유합 부전 |\n'
 '| 코드 801 | 특정 질병 근골격계질환 | 슬관절 탈구 / 아탈구 |\n'
 '| 코드 801 | 특정 질병 근골격계질환 | 십자인대손상 / 파열 |'),
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
 'indexing': {'chunk_id': 'chunk_001032',
              'chunk_char_len': 291,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)

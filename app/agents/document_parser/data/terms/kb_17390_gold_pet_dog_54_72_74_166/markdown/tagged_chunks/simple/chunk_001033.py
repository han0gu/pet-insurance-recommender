from langchain_core.documents import Document

chunk = Document(
    page_content=('| 코드 801 | 특정 질병 근골격계질환 | 십자인대손상 / 파열 |\n'
 '| 코드 801 | 특정 질병 근골격계질환 | 연골형성부전 퇴행성 관절질환 |\n'
 '| 코드 801 | 특정 질병 근골격계질환 | 골육종 |\n'
 '| 802 | 비뇨기계질환 | 방광게실 |\n'
 '| 802 | 비뇨기계질환 | 방광결석 |\n'
 '| 802 | 비뇨기계질환 | 방광염 |\n'
 '| 802 | 비뇨기계질환 | 사구체 신염 |\n'
 '| 802 | 비뇨기계질환 | 선천성 비뇨기 질환 |\n'
 '| 802 | 비뇨기계질환 | 수신증 |\n'
 '| 802 | 비뇨기계질환 | 신부전 신우신염 |'),
    metadata={'source_doc': {'total_pages': 113},
 'doc': {'doc_type': 'terms',
         'file_name': 'kb_17390_gold_pet_dog_54_72_74_166.pdf',
         'insurer_code': 'kb',
         'product_code': '17390',
         'product_name': '[일반보험] KB반려행복펫보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'other', 'risk_domains': ['joint', 'urinary']},
 'indexing': {'chunk_id': 'chunk_001033',
              'chunk_char_len': 293,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)

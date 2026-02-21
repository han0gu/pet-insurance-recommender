from langchain_core.documents import Document

chunk = Document(
    page_content=('- 나. 직업이 없는 자가 취직한 경우\n'
 '- 다. 현재의 직업을 그만둔 경우\n'
 '부 가 설 명 직업 또는 직무∙ 직업1) 생계유지 등을 위하여 일정한 기간동안(예: 6개월 이상) 계속하여 종사하는 일을 말합니다.2) '
 '1)에 해당하지 않는 경우에는 개인의 사회적 신분에 따르는 위치나 자리를 말합니다.예) 학생, 미취학아동, 무직 등∙ 직무직책이나 직업상 '
 '책임을 지고 담당하여 맡은 일을 말합니다.- 2. 보험증권 등에 기재된 피보험자의 운전 목적이 변경된 경우\n'
 '- 예) 자가용에서 영업용으로 변경, 영업용에서 자가용으로 변경 등\n'
 '- 법'),
    metadata={'source_doc': {'total_pages': 113},
 'doc': {'doc_type': 'terms',
         'file_name': 'kb_17390_gold_pet_dog_54_72_74_166.pdf',
         'insurer_code': 'kb',
         'product_code': '17390',
         'product_name': '[일반보험] KB반려행복펫보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'other', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000062',
              'chunk_char_len': 296,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)

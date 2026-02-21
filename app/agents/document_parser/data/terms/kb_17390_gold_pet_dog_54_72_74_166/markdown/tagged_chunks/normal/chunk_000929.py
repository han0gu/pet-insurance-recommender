from langchain_core.documents import Document

chunk = Document(
    page_content=('예방적으로 장기를 절제, 적출한 경우는 장해로 보지 않는다.\n'
 '7) 상기 흉복부 및 비뇨생식기계 장해항목에 명기되지 않은 기타 장해상태\n'
 '에 대해서는 ‘<붙임> 일상생활 기본동작(ADLs) 제한 장해평가표’에\n'
 '해당하는 장해가 있을 때 ADLs 장해 지급률을 준용한다.\n'
 '8) 상기 장해항목에 해당되지 않는 장기간의 간병이 필요한 만성질환(만성간\n'
 '질환, 만성폐쇄성폐질환 등)은 장해의 평가 대상으로 인정하지 않는다.- \n'
 '- \n'
 '# 13.# 신경계․정신행동 장해| 가. 장해의 분류 | 공 통 |\n'
 '| --- | --- |'),
    metadata={'source_doc': {'total_pages': 113},
 'doc': {'doc_type': 'terms',
         'file_name': 'kb_17390_gold_pet_dog_54_72_74_166.pdf',
         'insurer_code': 'kb',
         'product_code': '17390',
         'product_name': '[일반보험] KB반려행복펫보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'other',
            'risk_domains': ['digestive', 'head', 'urinary']},
 'indexing': {'chunk_id': 'chunk_000929',
              'chunk_char_len': 285,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)

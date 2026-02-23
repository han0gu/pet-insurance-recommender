from langchain_core.documents import Document

chunk = Document(
    page_content=('- 하나에 해당하는 때를 말한다.\n'
 '- 가) 폐, 신장, 또는 간장의 장기이식을 한 경우\n'
 '- 나) 장기이식을 하지 않고서는 생명유지가 불가능하여 혈액투석, 복막투\n'
 '- 석 등 의료처치를 평생토록 받아야 할 때\n'
 '- 다) 방광의 저장기능과 배뇨기능을 완전히 상실한 때\n'
 '3) ‘흉복부장기 또는 비뇨생식기 기능에 심한 장해를 남긴 때’라 함은 아\n'
 '래의 경우 중 하나에 해당하는 때를 말한다.\n'
 '가) 위, 대장(결장∼직장) 또는 췌장의 전부를 잘라내었을 때나) 소장을 3/4 이상 잘라내었을 때 또는 잘라낸 소장의 길이가 3m '
 '이상\n'
 '일 때'),
    metadata={'source_doc': {'total_pages': 113},
 'doc': {'doc_type': 'terms',
         'file_name': 'kb_17390_gold_pet_dog_54_72_74_166.pdf',
         'insurer_code': 'kb',
         'product_code': '17390',
         'product_name': '[일반보험] KB반려행복펫보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'other', 'risk_domains': ['digestive', 'urinary']},
 'indexing': {'chunk_id': 'chunk_000925',
              'chunk_char_len': 293,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)

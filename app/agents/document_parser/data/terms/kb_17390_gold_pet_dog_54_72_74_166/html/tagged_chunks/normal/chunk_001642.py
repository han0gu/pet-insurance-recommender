from langchain_core.documents import Document

chunk = Document(
    page_content=('한쪽 폐 또는 한쪽 신장을 전부 잘라내었을 때<br>나) 방광 기능상실로 영구적인 요도루, 방광루, 요관 장문합 상태<br>다) 위, '
 '췌장을 50% 이상 잘라내었을 때<br>라) 대장절제, 항문 괄약근 등의 기능장해로 영구적으로 장루, 인공항<br>문을 설치한 '
 '경우(치료과정에서 일시적으로 발생하는 경우는 제외)<br>마) 심장기능 이상으로 인공심박동기를 영구적으로 삽입한 경우<br>바) '
 "요도괄약근 등의 기능장해로 영구적으로 인공요도괄약근을 설치한<br>경우</p><p id='173' "
 "data-category='paragraph'"),
    metadata={'source_doc': {'total_pages': 113},
 'doc': {'doc_type': 'terms',
         'file_name': 'kb_17390_gold_pet_dog_54_72_74_166.pdf',
         'insurer_code': 'kb',
         'product_code': '17390',
         'product_name': '[일반보험] KB반려행복펫보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'exclusion',
            'risk_domains': ['digestive', 'urinary']},
 'indexing': {'chunk_id': 'chunk_001642',
              'chunk_char_len': 298,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.85}},
)

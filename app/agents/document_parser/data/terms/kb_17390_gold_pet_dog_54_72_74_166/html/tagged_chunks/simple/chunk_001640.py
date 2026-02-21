from langchain_core.documents import Document

chunk = Document(
    page_content=('‘흉복부장기 또는 비뇨생식기 기능에 심한 장해를 남긴 때’라 함은 아<br>래의 경우 중 하나에 해당하는 때를 말한다.<br>가) 위, '
 "대장(결장∼직장) 또는 췌장의 전부를 잘라내었을 때</p><br><p id='169' data-category='paragraph' "
 "style='font-size:14px'>나) 소장을 3/4 이상 잘라내었을 때 또는 잘라낸 소장의 길이가 3m 이상<br>일 "
 '때<br>다) 간장의 3/4 이상을 잘라내었을 때<br>라) 양쪽 고환 또는 양쪽 난소를 모두 잃었을 때<br>‘흉복부장기 또는 '
 '비뇨생식기'),
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
 'indexing': {'chunk_id': 'chunk_001640',
              'chunk_char_len': 298,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)

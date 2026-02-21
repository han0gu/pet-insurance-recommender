from langchain_core.documents import Document

chunk = Document(
    page_content=('눈을 뜨고 10m 거리를 직선으로 걸을 때 중앙에서 60cm 이상 벗어나는 '
 '경우</td><td>8</td></tr><tr><td>2)</td><td>평형기능의 장해는 장해판정 직전 1년 이상 지속적인 '
 "치료</td><td>후 장해가 고</td></tr></tbody></table><br><p id='196' "
 "data-category='paragraph' style='font-size:14px'>착되었을 때 판정하며, 뇌병변 여부, 전정기능 "
 '이상 및 장해상태를 평가<br>하기 위해 아래의 검사들을 기초로 한다.<br>가)'),
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
            'risk_domains': ['digestive', 'eye', 'head']},
 'indexing': {'chunk_id': 'chunk_001506',
              'chunk_char_len': 292,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)

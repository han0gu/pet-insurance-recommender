from langchain_core.documents import Document

chunk = Document(
    page_content=('. 평형기능의 장해<br>1) ‘평형기능에 장해를 남긴 때’라 함은 전정기관 이상으로 보행 등 일상<br>생활이 어려운 상태로 아래의 '
 "평형장해 평가항목별 합산점수가 30점 이</p><br><table id='195' "
 "style='font-size:14px'><thead><tr><td>상인</td><td>경우를 "
 '말한다.</td><td></td></tr></thead><tbody><tr><td rowspan="3"></td><td>항목 내 용 '
 '검사</td><td>점수</td></tr><tr><td>양측 전정기능'),
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
 'indexing': {'chunk_id': 'chunk_001503',
              'chunk_char_len': 286,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)

from langchain_core.documents import Document

chunk = Document(
    page_content=("받은 경우 1일 1회에 한하여 이 특별약관의</p><br><table id='75' "
 'style=\'font-size:14px\'><thead><tr><td colspan="2">가입금액을 창상봉합술 '
 '치료비로</td><td>지급합니다.</td></tr></thead><tbody><tr><td>창상봉합술Ⅰ (안면/경부)</td><td>구 '
 "분 상해로 '창상봉합술(안면/경부) 대상 수가코드'에 서 정한 '창상봉합술Ⅰ(급</td><td>지급금액 '창상봉합술 치료비Ⅰ "
 '(안면/경부)(1일1회한, 연간3회한,'),
    metadata={'source_doc': {'total_pages': 113},
 'doc': {'doc_type': 'terms',
         'file_name': 'kb_17390_gold_pet_dog_54_72_74_166.pdf',
         'insurer_code': 'kb',
         'product_code': '17390',
         'product_name': '[일반보험] KB반려행복펫보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'other', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000578',
              'chunk_char_len': 277,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)

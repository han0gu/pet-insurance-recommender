from langchain_core.documents import Document

chunk = Document(
    page_content=('계산한 지급보험금의 합계액 약<br>\uf000 피보험자가 다른 계약에 대하여 보험금 청구를 포기한 경우에도 회사의 제1항에 '
 "관</p><br><p id='123' data-category='paragraph' "
 "style='font-size:14px'>특</p><table id='124' "
 "style='font-size:16px'><thead></thead><tbody><tr><td>의한 지급보험금 "
 '결정에는</td><td>영향을 미치지 않습니다.</td></tr><tr><td colspan="2">용 어 풀 이 공제계약 유사보험으로서 '
 '공제'),
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
 'indexing': {'chunk_id': 'chunk_001122',
              'chunk_char_len': 297,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)

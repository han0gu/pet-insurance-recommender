from langchain_core.documents import Document

chunk = Document(
    page_content=('사실을 알지 못하였다는 사유로 회사에 이의를 제기할 수 없습니다.<br>\uf000 타인을 위한 계약에서 보험사고가 발생한 경우에 '
 '계약자가 그 타인에게 보험사고 상<br>의 발생으로 생긴 손해를 배상한 때에는 계약자는 그 타인의 권리를 해하지 않는 '
 "해</p><br><p id='42' data-category='list'></p><br><table id='43' "
 "style='font-size:16px'><thead></thead><tbody><tr><td>범위 안에서 회사에</td><td>보험금의 "
 '지급을 청구할 수'),
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
 'indexing': {'chunk_id': 'chunk_001236',
              'chunk_char_len': 284,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)

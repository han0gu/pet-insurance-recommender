from langchain_core.documents import Document

chunk = Document(
    page_content=('보험자의 증가, 감소 또는 교체) 제2항에도 불구하고 이 특별약관에 따라 보험료를 정산합니다.# 제2조(보험가입금액)상품다수구매자단체계약 '
 '특별약관 제3조(보험가입금액)의 규정에 관계없이 계약자가 피보험자의 보험가\n'
 '입금액을 각기 달리하여 가입하고자 할 경우에 회사는 계약사항을 고려하여 이를 승인할 수 있습니다.# 제3조(피보험자의 통지)- ① '
 '계약자는 피보험자의 증감이 있을 경우 아래 [양식1]에 정한 양식으로 회사에 서면(팩시밀리를 포\n'
 '- 함합니다)통지하여야 합니다.'),
    metadata={'source_doc': {'total_pages': 45},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_1_dog_anypet_3_20_21_47.pdf',
         'insurer_code': 'samsung',
         'product_code': '1',
         'product_name': '(일반)반려견보험 애니펫',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'other', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000163',
              'chunk_char_len': 263,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)

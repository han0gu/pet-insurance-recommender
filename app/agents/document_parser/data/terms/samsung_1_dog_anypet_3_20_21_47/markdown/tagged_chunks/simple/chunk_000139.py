from langchain_core.documents import Document

chunk = Document(
    page_content=('- 한 때에는 지정계좌에서 제1회 보험료를 받고 보험증권을 교부합니다.\n'
 '# 제2조(계약 후의 알릴 의무)계약자 또는 피보험자는 지정계좌의 번호가 변경 또는 거래정지된 경우에는 이 사실을 즉시 회사에\n'
 '알려야 합니다.# 제3조(준용규정)이 특별약관에 정하지 않은 사항은 보통약관을 따릅니다.- 34 -당신에게 좋은보험 삼성화재# 단체계약 '
 '특별약관# 제1조(계약의 적용 범위)① 피보험자가 다음 중 한가지의 단체에 소속되어야 하며, 단체를 대표하여 계약자로 된 자가 단체보'),
    metadata={'source_doc': {'total_pages': 45},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_1_dog_anypet_3_20_21_47.pdf',
         'insurer_code': 'samsung',
         'product_code': '1',
         'product_name': '(일반)반려견보험 애니펫',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'other', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000139',
              'chunk_char_len': 261,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)

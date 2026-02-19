from langchain_core.documents import Document

chunk = Document(
    page_content=('【수술비용 확대보장 보험금】 아래 ①과 ② 중 적은 금액\n'
 '(피보험자가 부담한 수술당일치료비 - 보통약관에서 지급한 치료비보험금) × 보상비율 ② 보험증권에서 정한 1일당 수술비용 확대보장 '
 '보상한도액\n'
 '② 보험기간 중에 발생한 사고로 회사가 지급하는 수술비용 확대보장 보험금의 총 한도는 보험증권에 기재된 총보상횟수를 한도로 합니다.\n'
 '제3조(준용규정)\n'
 '이 특별약관에 정하지 않은 사항은 보통약관을 따릅니다.'),
    metadata={'source_doc': {'total_pages': 45},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_1_dog_anypet_3_20_21_47.pdf',
         'insurer_code': 'samsung',
         'product_code': '1',
         'product_name': '(일반)반려견보험 애니펫',
         'total_pages': 45,
         'page': 22},
 'term_type': 'special',
 'clause': {'clause_type': 'limit', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000123',
              'chunk_char_len': 225,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'rule',
              'tag_confidence': 0.8}},
)

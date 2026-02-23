from langchain_core.documents import Document

chunk = Document(
    page_content=('을 원인으로 하여 생긴 반려동물의 치료비를 보통약관 제4조(보상하는 손해)에 따라 보상하여 드\n'
 '립니다.# 제2조(준용규정)이 특별약관에 정하지 않은 사항은 보통약관을 따릅니다.- 22 -당신에게 좋은보험 삼성화재# 반려동물 '
 '사망위로금 특별약관# 제1조(보상하는 손해)- ① 회사는 보험증권에 기재된 반려동물이 보험기간 중에 사망한 경우 보험증권에 기재된 '
 '보험가입금\n'
 '- 액을 보상하여 드립니다.\n'
 '- ② 제1항의 사망은 동물병원에서 적법하게 시행된 안락사를 포함합니다. 단, 이 경우 동물병원에서 발\n'
 '- 급한 소견서를 제출하여야 합니다.'),
    metadata={'source_doc': {'total_pages': 35},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_2_cat_anypet_3_20_21_37.pdf',
         'insurer_code': 'samsung',
         'product_code': '2',
         'product_name': '(일반)반려묘보험 애니펫',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'other', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000094',
              'chunk_char_len': 298,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)

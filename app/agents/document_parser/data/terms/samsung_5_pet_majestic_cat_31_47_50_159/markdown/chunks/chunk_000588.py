from langchain_core.documents import Document

chunk = Document(
    page_content=('하지 않은 사항은 특별약관 일반사항을 따릅니다. 특별약관 일반사항에서도 정하지 않은\n'
 '사항은 보통약관을 따릅니다. 다만, 보통약관 제10조(환급금의 중도인출), 제11조(만기환\n'
 '급금의 지급)은 제외합니다.- 112 -4-4. 반려묘 의료비 확대보장(이물제거 특정처치)(연간2회한)\n'
 '(재가입형) 특별약관# 제1조 (보험금의 지급사유)- ① 회사는 보험증권에 기재된 이 특별약관의 보험기간(이하 「보험기간」 이라 합니다) '
 '중\n'
 '- 에 보험증권에 기재된 반려묘가 국내에서 수의사에게 이물 섭취 치료를 목적으로 이'),
    metadata={'source_doc': {'total_pages': 129},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_5_pet_majestic_cat_31_47_50_159.pdf',
         'insurer_code': 'samsung',
         'product_code': '5',
         'product_name': '(장기)무배당 삼성화재 펫보험 의기냥냥',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown'},
)

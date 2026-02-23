from langchain_core.documents import Document

chunk = Document(
    page_content=('항을 따릅니다. 특별약관 일반사항에서도 정하지 않은 사항은 보통약관을 따릅니다. 다만\n'
 ', 보통약관 제10조(환급금의 중도인출), 제11조(만기환급금의 지급)은 제외합니다.- 80 -80 / 1813-3. 반려견 '
 '의료비(치과및구강질환포함)(수술당일)(재가입형) 특별약관# 제 1조 (보험금의 지급사유)- ① 회사는 보험증권에 기재된 이 특별약관의 '
 '보험기간(이하「보험기간」이라 합니다) 중\n'
 '- 에 제3항에서 정한 보장개시일(책임개시일) 이후에 보험증권에 기재된 반려견에게 상'),
    metadata={'source_doc': {'total_pages': 78},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_3_direct_good_pet_28_42_45_105.pdf',
         'insurer_code': 'samsung',
         'product_code': '3',
         'product_name': '(장기)무배당 삼성화재 다이렉트 '
                         '착한펫보험(강아지)',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown'},
)

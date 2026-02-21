from langchain_core.documents import Document

chunk = Document(
    page_content=('<용어풀이>[습관성 유산, 불임 및 인공수정 관련 합병증]\n'
 '한국표준질병·사인분류상의 N96~N98에 해당하는 질병을 말합니다.ㅜ!ㅜ!! OI 그의 〔 거레 I I ニ」 CH I IIコー I '
 'I열거된 행위로 인하여 각 특별약관별 보험금의 지급사유의 보험금 지급사유가 발생한\n'
 '때에는 해당 보험금을 지급하지 않습니다.- 1. 전문등반(전문적인 등산용구를 사용하여 암벽 또는 빙벽을 오르내리거나 특수한 기\n'
 '- 술, 경험, 사전훈련을 필요로 하는 등반을 말합니다), 글라이더 조종, 스카이다이'),
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

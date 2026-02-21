from langchain_core.documents import Document

chunk = Document(
    page_content=('합니다) 중에 상해 또는 진단확정된 질병으로 아래에 정한 창상봉합술을 받은 경우 각 보\n'
 '장별 1일 1회, 총 연간 3회에 한하여 아래에 기재된 지급금액을 창상봉합술 치료비(1일1\n'
 '회한) (이하 「창상봉합술 치료비」 라 합니다)로 보험수익자에게 지급합니다.| 보장구분 | 보험금 지급사유 | 지급금액 |\n'
 '| --- | --- | --- |\n'
 '| 창상봉합술 (3/5cm 미만,급여)(A) | 상해 및질병으로 제3조(창상봉합술의 정의와 장소)에서 정한 '
 '「창상봉합술(3/5cm미만,급여)」 을 받는 경우 | 이 특별약관 가입금액의 10% |'),
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

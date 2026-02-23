from langchain_core.documents import Document

chunk = Document(
    page_content=('- 경우 각막이식술 이전의 시력상태를 기준으로 평가한다.\n'
 '- 3) "한 눈이 멀었을 때" 라 함은 안구의 적출은 물론 명암을 가리지 못하거나\n'
 '- ( "광각무" ) 겨우 가릴 수 있는 경우( "광각유" )를 말한다.\n'
 '- 4) "한눈의 교정시력이 0.02 이하로 된 때" 라 함은 안전수동(Hand Movement)주\n'
 '- 1) 안전수지(Finger Counting)주2) 상태를 포함한다.\n'
 '※ 주1) 안전수동 : 물체를 감별할 정도의 시력상태가 아니며 눈앞에서 손의 움직임을 식\n'
 '별할 수 있을 정도의 시력상태'),
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

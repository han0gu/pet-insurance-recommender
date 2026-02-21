from langchain_core.documents import Document

chunk = Document(
    page_content=('| 안면부 창상봉합술 (단순봉합 제외,급여)(D) | 상해 및 질병으로 제3조(창상봉합술의 정의와 장소)에서 정한 「안면부 '
 '창상봉합술(단순봉합 제외,급여)」 을 받는 경우 | 이 특별약관 가입금액의 100% |\n'
 '| 보장구분 | 지급기준 | 지급기준 |\n'
 '| --- | --- | --- |\n'
 '| 보장구분 | 안면 또는 경부 | 안면과 경부 이외 |\n'
 '| 창상봉합술 (3/5cm 미만,급여)(A) | 표재성 3cm 미만 | 표재성 5cm 미만 |\n'
 '| 보장구분 | 지급기준 | 지급기준 |\n'
 '| --- | --- | --- |'),
    metadata={'source_doc': {'total_pages': 107},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_4_pet_majestic_dog_33_49_52_139.pdf',
         'insurer_code': 'samsung',
         'product_code': '4',
         'product_name': '(장기)무배당 삼성화재 펫보험 위풍댕댕',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown'},
)

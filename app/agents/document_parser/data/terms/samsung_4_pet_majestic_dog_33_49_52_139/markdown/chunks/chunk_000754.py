from langchain_core.documents import Document

chunk = Document(
    page_content=('경 지정할 수 있습니다. 이 경우 회사는 변경 지정을 서면으로 알리거나 보험증권의\n'
 '뒷면에 기재하여 드립니다.- 1. 지정대리청구인 변경신청서(회사양식)\n'
 '- 2. 지정대리청구인의 가족관계등록부(기본증명서 등)\n'
 '- 3. 신분증(주민등록증이나 운전면허증 등 사진이 붙은 정부기관 발행 신분증, 본인이\n'
 '- 아닌 경우에는 본인의 인감증명서, 본인서명사실확인서 또는 안전성과 신뢰성이\n'
 '- 확보된 전자적 수단을 활용한 계약자 의사표시의 확인방법 포함)\n'
 '- ② 제1항에도 불구하고 보험계약에서 지정대리청구인의 지정 기간을 별도로 제한한 경'),
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

from langchain_core.documents import Document

chunk = Document(
    page_content=('니다.- \n'
 '# 제4조 (지정대리청구인의 변경지정)① 계약자는 이 특별약관의 계약체결 이후 다음의 서류를 제출하고 지정대리청구인을 변경 지정할 수 '
 '있습니다. 이 경우 회사는 변경 지정을 서면으로 알리거나 보험증권의# 뒷면에 기재하여 드립니다.- 1. 지정대리청구인 '
 '변경신청서(회사양식)\n'
 '- 2. 지정대리청구인의 가족관계등록부(기본증명서 등)\n'
 '- 3. 신분증(주민등록증이나 운전면허증 등 사진이 붙은 정부기관 발행 신분증, 본인이\n'
 '- 아닌 경우에는 본인의 인감증명서, 본인서명사실확인서 또는 안전성과 신뢰성이'),
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

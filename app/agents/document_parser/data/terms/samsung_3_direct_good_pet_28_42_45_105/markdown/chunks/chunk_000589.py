from langchain_core.documents import Document

chunk = Document(
    page_content=('- 여야 하며, 보험계약의 보험료를 선납하는 경우에도 또한 같습니다.\n'
 '제5조 (특별약관 내용의 변경)- 104 -104 / 181# 이 특별약관이 부가된 보험계약의 경우에는 보험계약 약관의 규정에도 불구하고 '
 '다음과\n'
 '같은 내용은 변경할 수 없습니다.1. 보험기간의 변경\n'
 '2. 감액완납보험으로의 변경- \n'
 '<용어풀이>[감액완납보험]차회 이후의 보험료 납입을 중단하는 대신 가입금액을 감액하는 보험제6조 (준용규정)이 특별약관에 정하지 않은 '
 '사항은 보험계약을 따릅니다.- 105 -105 / 181'),
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

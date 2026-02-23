from langchain_core.documents import Document

chunk = Document(
    page_content=('만기시 자동갱신되는 보험계약의 경우 자동갱신의 조건<br>- 저축성 보험계약의 공시이율<br>- 유배당 보험계약의 경우 계약자 배당에 '
 "관한 사항<br>- 그 밖에 약관에 기재된 보험계약의 중요사항</p><br><p id='87' "
 "data-category='paragraph' style='font-size:16px'>\uf000 제1항과 관련하여 통신판매계약의 "
 '경우, 회사는 계약자<br>가 가입한 특별약관만 포함한 약관을 드리며, 전화를 이용<br>하여 체결하는 계약은 계약자의 동의를 얻어 '
 '다음의 방법으<br>로 약관의 중요한 내용을 설명할'),
    metadata={'source_doc': {'total_pages': 160},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_2_petpermint_cat_family_45_82_83_206.pdf',
         'insurer_code': 'meritz',
         'product_code': '2',
         'product_name': '무배당 펫퍼민트 Cat&Family보험 다이렉트2601',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown'},
)

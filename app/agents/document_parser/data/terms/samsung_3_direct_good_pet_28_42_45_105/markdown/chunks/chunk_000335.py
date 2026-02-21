from langchain_core.documents import Document

chunk = Document(
    page_content=('우에는 우편, 전화, 방문 등의 방법으로 지체없이 회사에 알려야 합니다.- 1. 청약서의 기재사항을 변경하고자 할 때 또는 변경이 '
 '생겼음을 알았을 때\n'
 '- 2. 이 특별약관에서 보장하는 위험과 동일한 위험을 보장하는 계약을 다른 보험자와\n'
 '- 체결하고자 할 때 또는 이와 같은 계약이 있음을 알았을 때\n'
 '- 3. 반려견을 양도할 때\n'
 '- 4. 위 이외에 위험이 뚜렷이 변경되거나 변경되었음을 알았을 때\n'
 '② 회사는 제1항의 통지로 인하여 위험의 변동이 발생한 경우에는 보통약관 제21조(계약'),
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

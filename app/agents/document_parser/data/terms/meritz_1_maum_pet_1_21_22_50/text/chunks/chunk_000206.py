from langchain_core.documents import Document

chunk = Document(
    page_content=('애인 증명서」라 합니다)을 제출하여 제1조(특별약관의 적용범위) 제1항 제2호에서 정\n'
 '한 조건에 해당함을 회사에 알려야 합니다.② 제1항에도 불구하고「국가유공자 등 예우 및 지원에 관한 법률」에 따른 상이자의 증\n'
 '명을 받은 사람 또는「장애인복지법」에 따른 장애인등록증을 발급받은 사람에 대해서\n'
 '는 해당 증명서·장애인등록증의 사본이나 그 밖의 장애 사실을 증명하는 서류를 제출하\n'
 '는 경우에는 제 1항의 장애인증명서는 제출하지 않을 수 있습니다.\n'
 '③ 장애인으로서 그 장애기간이 기재된 장애인증명서를 제1항 따라 회사에 제출한 때에'),
    metadata={'source_doc': {'total_pages': 50},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_1_maum_pet_1_21_22_50.pdf',
         'insurer_code': 'meritz',
         'product_code': '1',
         'product_name': '메리츠 마음든든 반려동물보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown'},
)

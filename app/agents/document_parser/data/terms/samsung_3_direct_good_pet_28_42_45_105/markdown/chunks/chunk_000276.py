from langchain_core.documents import Document

chunk = Document(
    page_content=('- 보통약관에서 보험료 납입면제에 대해 정하지 않은 경우 특별약관 일반사항 제5조(보\n'
 '- 험료 납입면제) 및 제6조(보험료 납입면제에 관한 세부규정)를 적용하지 않습니다.\n'
 '- ③ 제1항에도 불구하고, 특별약관 일반사항 제35조(해약환급금)를 적용하지 않으며 보통\n'
 '- 약관 제33조(해약환급금)을 적용합니다.\n'
 '# 제2관 개별사항# 제1조 (보험금의 지급사유)회사는 피보험자가 보험증권에 기재된 이 특별약관의 보험기간(이하「보험기간」이라 합\n'
 '니다) 중에 질병으로 사망한 경우 5년간 매월 보험증권에 기재된 이 특별약관의 보험가'),
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

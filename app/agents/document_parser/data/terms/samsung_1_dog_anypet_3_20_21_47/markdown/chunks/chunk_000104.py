from langchain_core.documents import Document

chunk = Document(
    page_content=('- ② 보험증권에서 정한 1일당 보상한도액\n'
 '# 제3조(준용규정)이 특별약관에 정하지 않은 사항은 보통약관을 따릅니다.- 24 -당신에게 좋은보험 삼성화재# 배상책임보장 '
 '특별약관제1조(보상하는 손해)① 회사는 피보험자가 국내(북한지역 제외)에서 보험기간 중에 보험증권에 기재된 피보험자의 가입동\n'
 '물의 행위에 기인하는 우연한 사고(이하 "사고"라 합니다.)로 인하여 피해자의 신체에 장해(상해,\n'
 '질병 및 그로 인한 사망을 말합니다.)를 입히거나 피해자 소유의 동물에 손해를 입혀 법률상의 배'),
    metadata={'source_doc': {'total_pages': 45},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_1_dog_anypet_3_20_21_47.pdf',
         'insurer_code': 'samsung',
         'product_code': '1',
         'product_name': '(일반)반려견보험 애니펫',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown'},
)

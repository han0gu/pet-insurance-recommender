from langchain_core.documents import Document

chunk = Document(
    page_content=('- 을 경우\n'
 '\uf000 계약자 또는 피보험자가 제1항 각 호의 통지를 게을리하\n'
 '여 손해가 증가된 때에는 회사는 그 증가된 손해는 보상하\n'
 '지 않으며, 제1항 제3호의 통지를 게을리 한 때에는 소송비\n'
 '용과 변호사비용도 보상하지 않습니다. 다만, 계약자 또는\n'
 '피보험자가 상법 제657조 제1항에 의해 보험사고의 발생을\n'
 '회사에 알린 경우에는 계약자 또는 피보험자가 지출한 아래\n'
 '의 손해 및 비용에 대하여 보상한도액을 한도로 보상합니\n'
 '다.- ① 피보험자가 피해자에게 지급할 책임을 지는 법률상의\n'
 '- 손해배상금(손해배상금을 지급함으로써 대위 취득할'),
    metadata={'source_doc': {'total_pages': 180},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_3_petpermint_dog_family_51_86_87_230.pdf',
         'insurer_code': 'meritz',
         'product_code': '3',
         'product_name': '무배당 펫퍼민트 Puppy&Family보험 다이렉트2601',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown'},
)

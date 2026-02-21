from langchain_core.documents import Document

chunk = Document(
    page_content=('| 1 | 뒷다리 근골격계 질환 | NAA003 | 고관절 (아) 탈구 (좌측) |\n'
 '| 1 | 뒷다리 근골격계 질환 | NAA004 | 고관절 (아) 탈구 (우측) |\n'
 '| 1 | 뒷다리 근골격계 질환 | NAA005 | 무혈성골두괴사(LCPD) (좌측) |\n'
 '| 1 | 뒷다리 근골격계 질환 | NAA006 | 무혈성골두괴사(LCPD) (우측) |\n'
 '| 1 | 뒷다리 근골격계 질환 | NAA007 | 슬개골 (아) 탈구- (좌측-1기) |\n'
 '| 1 | 뒷다리 근골격계 질환 | NAA008 | 슬개골 (아) 탈구- (좌측-2,3,4기) |'),
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

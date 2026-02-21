from langchain_core.documents import Document

chunk = Document(
    page_content=('| 1 | 뒷다리 근골격계 질환 | NAA009 | 슬개골 (아) 탈구- (우측-1기) |\n'
 '| 1 | 뒷다리 근골격계 질환 | NAA010 | 슬개골 (아) 탈구- (우측-2,3,4기) |\n'
 '| 1 | 뒷다리 근골격계 질환 | NAA011 NAA012 | 십자 인대 손상 파열 (전방 / 후방) (좌측) 십자 인대 손상 파열 '
 '(전방 / 후방) (우측) |\n'
 '| 1 | 뒷다리 근골격계 질환 | NAA013 | 골절 (뒷다리) (좌측) |\n'
 '| 1 | 뒷다리 근골격계 질환 | NAA014 | 골절 (뒷다리) (우측) |'),
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
